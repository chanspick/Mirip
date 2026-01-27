// react-router-dom 수동 모킹
// Jest가 모듈 해석 이슈가 있을 때 사용

import React from 'react';

const mockNavigate = jest.fn();

export const useNavigate = () => mockNavigate;
export const useLocation = () => ({ pathname: '/' });
export const useParams = () => ({});

export const Link = ({ children, to, className, ...props }) => (
  <a href={to} className={className} {...props}>
    {children}
  </a>
);

export const BrowserRouter = ({ children }) => <div>{children}</div>;
export const MemoryRouter = ({ children }) => <div>{children}</div>;
export const Routes = ({ children }) => <div>{children}</div>;
export const Route = ({ element }) => element;

export default {
  useNavigate,
  useLocation,
  useParams,
  Link,
  BrowserRouter,
  MemoryRouter,
  Routes,
  Route,
};
